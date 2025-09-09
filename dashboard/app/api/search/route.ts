import { NextRequest, NextResponse } from 'next/server'
import { searchUsers, getUserById } from '@/lib/dataService'

export async function GET(request: NextRequest) {
  try {
    const { searchParams } = new URL(request.url)
    const searchTerm = searchParams.get('search') || ''
    const userType = searchParams.get('type') as 'Driver' | 'Merchant' || 'Driver'
    const userId = searchParams.get('userId')
    
    let results
    
    if (userId) {
      // Get specific user
      const user = await getUserById(parseInt(userId), userType)
      results = user ? [user] : []
    } else {
      // Search users
      results = await searchUsers(searchTerm, userType)
    }

    return NextResponse.json({
      success: true,
      data: results,
      count: results.length,
      searchTerm,
      userType
    })
  } catch (error) {
    console.error('Search API Error:', error)
    return NextResponse.json(
      { error: 'Failed to search users' },
      { status: 500 }
    )
  }
}

export async function POST(request: NextRequest) {
  try {
    const { searchTerm, userType, filters } = await request.json()
    
    const results = await searchUsers(searchTerm, userType)
    
    // Apply additional filters if provided
    let filteredResults = results
    if (filters) {
      if (filters.minCreditScore) {
        filteredResults = filteredResults.filter(user => user.credit_score >= filters.minCreditScore)
      }
      if (filters.maxCreditScore) {
        filteredResults = filteredResults.filter(user => user.credit_score <= filters.maxCreditScore)
      }
      if (filters.minRating) {
        filteredResults = filteredResults.filter(user => user.average_rating >= filters.minRating)
      }
      if (filters.location) {
        filteredResults = filteredResults.filter(user => user.location === filters.location)
      }
      if (filters.ageGroup) {
        filteredResults = filteredResults.filter(user => user.age_group === filters.ageGroup)
      }
    }

    return NextResponse.json({
      success: true,
      data: filteredResults,
      count: filteredResults.length,
      searchTerm,
      userType,
      filtersApplied: filters
    })
  } catch (error) {
    console.error('Advanced Search API Error:', error)
    return NextResponse.json(
      { error: 'Failed to perform advanced search' },
      { status: 500 }
    )
  }
}
